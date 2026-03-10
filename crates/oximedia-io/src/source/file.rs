//! File-based media source implementation.

use super::MediaSource;
use async_trait::async_trait;
use oximedia_core::{OxiError, OxiResult};
use std::io::SeekFrom;
use std::path::Path;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

/// A media source backed by a local file.
///
/// Uses tokio's async file I/O for non-blocking reads and seeks.
///
/// # Example
///
/// ```no_run
/// use oximedia_io::source::{FileSource, MediaSource};
///
/// #[tokio::main]
/// async fn main() -> oximedia_core::OxiResult<()> {
///     let mut source = FileSource::open("video.webm").await?;
///
///     println!("File size: {:?}", source.len());
///     println!("Position: {}", source.position());
///
///     let mut buffer = [0u8; 1024];
///     let n = source.read(&mut buffer).await?;
///     println!("Read {} bytes", n);
///
///     Ok(())
/// }
/// ```
pub struct FileSource {
    file: File,
    length: u64,
    position: u64,
    writable: bool,
}

impl FileSource {
    /// Creates a new `FileSource` from an already opened file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file metadata cannot be read.
    pub async fn new(file: File) -> OxiResult<Self> {
        let metadata = file.metadata().await?;
        let length = metadata.len();

        Ok(Self {
            file,
            length,
            position: 0,
            writable: false,
        })
    }

    /// Creates a new writable `FileSource` from an already opened file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file metadata cannot be read.
    pub async fn new_writable(file: File) -> OxiResult<Self> {
        let metadata = file.metadata().await?;
        let length = metadata.len();

        Ok(Self {
            file,
            length,
            position: 0,
            writable: true,
        })
    }

    /// Opens a file at the given path and creates a `FileSource`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or its metadata
    /// cannot be read.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oximedia_io::source::FileSource;
    ///
    /// #[tokio::main]
    /// async fn main() -> oximedia_core::OxiResult<()> {
    ///     let source = FileSource::open("video.webm").await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn open<P: AsRef<Path>>(path: P) -> OxiResult<Self> {
        let file = File::open(path).await?;
        Self::new(file).await
    }

    /// Creates a new file at the given path for writing.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oximedia_io::source::FileSource;
    ///
    /// #[tokio::main]
    /// async fn main() -> oximedia_core::OxiResult<()> {
    ///     let source = FileSource::create("output.webm").await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn create<P: AsRef<Path>>(path: P) -> OxiResult<Self> {
        let file = tokio::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .await?;
        Self::new_writable(file).await
    }
}

#[async_trait]
impl MediaSource for FileSource {
    async fn read(&mut self, buf: &mut [u8]) -> OxiResult<usize> {
        let n = self.file.read(buf).await?;
        self.position += n as u64;
        Ok(n)
    }

    async fn write_all(&mut self, buf: &[u8]) -> OxiResult<()> {
        if !self.writable {
            return Err(OxiError::unsupported("File is not open for writing"));
        }
        self.file.write_all(buf).await?;
        self.position += buf.len() as u64;
        if self.position > self.length {
            self.length = self.position;
        }
        Ok(())
    }

    async fn seek(&mut self, pos: SeekFrom) -> OxiResult<u64> {
        let new_pos = self.file.seek(pos).await?;
        self.position = new_pos;
        Ok(new_pos)
    }

    fn len(&self) -> Option<u64> {
        Some(self.length)
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

impl std::fmt::Debug for FileSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileSource")
            .field("length", &self.length)
            .field("position", &self.position)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_file_source_open_and_read() {
        // Create a temporary file with some content
        let mut temp_file = NamedTempFile::new().expect("failed to create temp file");
        let content = b"Hello, OxiMedia!";
        temp_file.write_all(content).expect("failed to write");
        temp_file.flush().expect("failed to flush");

        // Open the file source
        let mut source = FileSource::open(temp_file.path())
            .await
            .expect("failed to open file source");

        // Check length
        assert_eq!(source.len(), Some(content.len() as u64));
        assert!(source.is_seekable());

        // Read content
        let mut buffer = [0u8; 32];
        let n = source.read(&mut buffer).await.expect("failed to read");
        assert_eq!(n, content.len());
        assert_eq!(&buffer[..n], content);
    }

    #[tokio::test]
    async fn test_file_source_seek() {
        let mut temp_file = NamedTempFile::new().expect("failed to create temp file");
        temp_file.write_all(b"0123456789").expect("failed to write");
        temp_file.flush().expect("failed to flush");

        let mut source = FileSource::open(temp_file.path())
            .await
            .expect("failed to open file source");

        // Seek to position 5
        let pos = source
            .seek(SeekFrom::Start(5))
            .await
            .expect("seek should succeed");
        assert_eq!(pos, 5);
        assert_eq!(source.position(), 5);

        // Read from position 5
        let mut buffer = [0u8; 5];
        let n = source.read(&mut buffer).await.expect("failed to read");
        assert_eq!(n, 5);
        assert_eq!(&buffer, b"56789");
    }

    #[tokio::test]
    async fn test_file_source_nonexistent() {
        let result = FileSource::open("/nonexistent/path/to/file.webm").await;
        assert!(result.is_err());
    }
}
