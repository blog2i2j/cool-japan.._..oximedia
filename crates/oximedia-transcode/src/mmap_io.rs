//! Large-file I/O abstraction for memory-efficient transcoding.
//!
//! Reading large media files through the OS kernel page cache can reduce
//! memory pressure compared to fully buffered I/O.  This module provides a
//! pure-Rust layered I/O strategy that:
//!
//! - Uses large read-ahead buffers to amortise syscall cost.
//! - Supports random-access reads (seek + read) via [`LargeFileReader`].
//! - Provides a sliding-window view ([`FileWindow`]) over a sub-region.
//! - Tracks read statistics for profiling ([`ReadStats`]).
//!
//! The interface deliberately mirrors what a real `mmap`-backed reader would
//! expose so that callers can swap implementations without changing their code.
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_transcode::mmap_io::{LargeFileReader, LargeFileConfig};
//!
//! # fn example() -> std::io::Result<()> {
//! let mut reader = LargeFileReader::open("large_input.mxf", LargeFileConfig::default())?;
//! println!("file size: {} bytes", reader.file_len());
//!
//! // Random-access read of 1 KiB at offset 4096.
//! let mut buf = vec![0u8; 1024];
//! let n = reader.read_at(4096, &mut buf)?;
//! println!("read {} bytes", n);
//! # Ok(())
//! # }
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

// ─── Configuration ────────────────────────────────────────────────────────────

/// I/O strategy hint — affects the internal buffer size and prefetch policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// File is read front-to-back (default for transcoding pipelines).
    Sequential,
    /// File is accessed in an unpredictable order (container probing).
    Random,
    /// A mix of sequential runs and occasional seeks (typical MXF/MOV).
    Mixed,
}

impl Default for AccessPattern {
    fn default() -> Self {
        Self::Sequential
    }
}

/// Tuning parameters for [`LargeFileReader`].
#[derive(Debug, Clone)]
pub struct LargeFileConfig {
    /// Internal read-ahead buffer size in bytes.
    ///
    /// Larger values reduce syscall count at the cost of memory.
    /// Default: 4 MiB.
    pub buffer_size: usize,

    /// Expected access pattern — controls buffer strategy.
    pub access_pattern: AccessPattern,

    /// Maximum bytes to prefetch at open time (0 = no prefetch).
    ///
    /// Prefetching is implemented as a sequential read that fills the internal
    /// buffer, encouraging the OS to load pages into the page cache.
    pub prefetch_bytes: usize,
}

impl Default for LargeFileConfig {
    fn default() -> Self {
        Self {
            buffer_size: 4 * 1024 * 1024,
            access_pattern: AccessPattern::Sequential,
            prefetch_bytes: 16 * 1024 * 1024,
        }
    }
}

impl LargeFileConfig {
    /// Optimised for sequential large-file reading.
    #[must_use]
    pub fn sequential() -> Self {
        Self {
            buffer_size: 8 * 1024 * 1024,
            access_pattern: AccessPattern::Sequential,
            prefetch_bytes: 32 * 1024 * 1024,
        }
    }

    /// Optimised for random access (e.g. container probing).
    #[must_use]
    pub fn random_access() -> Self {
        Self {
            buffer_size: 512 * 1024,
            access_pattern: AccessPattern::Random,
            prefetch_bytes: 0,
        }
    }

    /// Sets the buffer size.
    #[must_use]
    pub fn with_buffer_size(mut self, bytes: usize) -> Self {
        self.buffer_size = bytes.max(1);
        self
    }

    /// Sets the access pattern hint.
    #[must_use]
    pub fn with_access_pattern(mut self, pattern: AccessPattern) -> Self {
        self.access_pattern = pattern;
        self
    }
}

// ─── Read statistics ──────────────────────────────────────────────────────────

/// Accounting accumulated by a [`LargeFileReader`].
#[derive(Debug, Clone, Default)]
pub struct ReadStats {
    /// Total bytes returned to callers.
    pub bytes_read: u64,
    /// Number of `read_at` / `read` calls issued.
    pub read_calls: u64,
    /// Number of seek operations issued.
    pub seek_count: u64,
    /// Largest single read requested.
    pub peak_read_bytes: u64,
}

impl ReadStats {
    /// Returns the mean bytes per read call, or `0.0` if no reads were made.
    #[must_use]
    pub fn mean_read_bytes(&self) -> f64 {
        if self.read_calls == 0 {
            0.0
        } else {
            self.bytes_read as f64 / self.read_calls as f64
        }
    }
}

// ─── LargeFileReader ─────────────────────────────────────────────────────────

/// A buffered random-access reader optimised for large media files.
///
/// Wraps a [`BufReader<File>`] with additional accounting and convenience
/// methods that mirror a memory-mapped interface.
pub struct LargeFileReader {
    inner: BufReader<File>,
    file_len: u64,
    path: PathBuf,
    stats: ReadStats,
    config: LargeFileConfig,
}

impl std::fmt::Debug for LargeFileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LargeFileReader")
            .field("path", &self.path)
            .field("file_len", &self.file_len)
            .finish()
    }
}

impl LargeFileReader {
    /// Opens a file and prepares the reader with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] when the file cannot be opened or its length
    /// cannot be determined.
    pub fn open(path: impl AsRef<Path>, config: LargeFileConfig) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let file_len = file.metadata()?.len();
        let buf_reader = BufReader::with_capacity(config.buffer_size, file);

        let mut reader = Self {
            inner: buf_reader,
            file_len,
            path,
            stats: ReadStats::default(),
            config,
        };

        // Prefetch: trigger a sequential read to warm the OS page cache.
        if reader.config.prefetch_bytes > 0 && file_len > 0 {
            let prefetch = (reader.config.prefetch_bytes as u64).min(file_len) as usize;
            let mut discard = vec![0u8; prefetch.min(reader.config.buffer_size)];
            // Read and immediately discard — the OS page cache retains the data.
            let _ = reader.inner.read(&mut discard);
            // Seek back to start.
            reader
                .inner
                .seek(SeekFrom::Start(0))
                .map_err(|e| io::Error::new(e.kind(), format!("prefetch seek: {e}")))?;
            reader.stats.seek_count += 1;
        }

        Ok(reader)
    }

    /// Opens a file using default configuration.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] when the file cannot be opened.
    pub fn open_default(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::open(path, LargeFileConfig::default())
    }

    /// Returns the total file size in bytes.
    #[must_use]
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Returns `true` if the file is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.file_len == 0
    }

    /// Returns the file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns a copy of the current read statistics.
    #[must_use]
    pub fn stats(&self) -> &ReadStats {
        &self.stats
    }

    /// Reads up to `buf.len()` bytes starting at absolute `offset`.
    ///
    /// The internal cursor is left at `offset + bytes_read` after the call.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] on seek or read failure.
    pub fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.seek(SeekFrom::Start(offset))?;
        self.stats.seek_count += 1;
        let n = self.inner.read(buf)?;
        let n_u64 = n as u64;
        self.stats.bytes_read += n_u64;
        self.stats.read_calls += 1;
        if n_u64 > self.stats.peak_read_bytes {
            self.stats.peak_read_bytes = n_u64;
        }
        Ok(n)
    }

    /// Reads exactly `buf.len()` bytes starting at `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::UnexpectedEof`] if fewer bytes are available.
    pub fn read_exact_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.inner.seek(SeekFrom::Start(offset))?;
        self.stats.seek_count += 1;
        self.inner.read_exact(buf)?;
        let n = buf.len() as u64;
        self.stats.bytes_read += n;
        self.stats.read_calls += 1;
        if n > self.stats.peak_read_bytes {
            self.stats.peak_read_bytes = n;
        }
        Ok(())
    }

    /// Reads a fixed-size array of `N` bytes at `offset` and returns it.
    ///
    /// Convenient for parsing box headers, magic bytes, etc.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] on seek/read failure or EOF.
    pub fn read_array_at<const N: usize>(&mut self, offset: u64) -> io::Result<[u8; N]> {
        let mut buf = [0u8; N];
        self.read_exact_at(offset, &mut buf)?;
        Ok(buf)
    }

    /// Returns `true` if `offset` is within the file.
    #[must_use]
    pub fn contains_offset(&self, offset: u64) -> bool {
        offset < self.file_len
    }

    /// Creates a [`FileWindow`] over the region `[offset, offset + len)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested region extends beyond the end of the
    /// file.
    pub fn window(&self, offset: u64, len: u64) -> io::Result<FileWindow> {
        let end = offset.checked_add(len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "FileWindow: offset + len overflow",
            )
        })?;
        if end > self.file_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "FileWindow [{offset}, {end}) exceeds file length {}",
                    self.file_len
                ),
            ));
        }
        Ok(FileWindow {
            path: self.path.clone(),
            file_offset: offset,
            len,
            cursor: 0,
        })
    }

    /// Reads a `u32` big-endian integer at `offset`.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] on read failure.
    pub fn read_u32_be(&mut self, offset: u64) -> io::Result<u32> {
        let bytes: [u8; 4] = self.read_array_at(offset)?;
        Ok(u32::from_be_bytes(bytes))
    }

    /// Reads a `u64` big-endian integer at `offset`.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] on read failure.
    pub fn read_u64_be(&mut self, offset: u64) -> io::Result<u64> {
        let bytes: [u8; 8] = self.read_array_at(offset)?;
        Ok(u64::from_be_bytes(bytes))
    }
}

impl Read for LargeFileReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        let n_u64 = n as u64;
        self.stats.bytes_read += n_u64;
        self.stats.read_calls += 1;
        if n_u64 > self.stats.peak_read_bytes {
            self.stats.peak_read_bytes = n_u64;
        }
        Ok(n)
    }
}

impl Seek for LargeFileReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.stats.seek_count += 1;
        self.inner.seek(pos)
    }
}

// ─── FileWindow ───────────────────────────────────────────────────────────────

/// A lazily-opened, seekable view over a sub-region of a file.
///
/// Each [`FileWindow`] opens its own file handle so that multiple windows
/// can be read independently without interfering with each other.
#[derive(Debug, Clone)]
pub struct FileWindow {
    path: PathBuf,
    /// Absolute byte offset of this window within the file.
    file_offset: u64,
    /// Length of this window in bytes.
    len: u64,
    /// Cursor relative to the window start.
    cursor: u64,
}

impl FileWindow {
    /// Returns the length of this window in bytes.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Returns `true` if the window is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the window's absolute byte offset within the file.
    #[must_use]
    pub fn file_offset(&self) -> u64 {
        self.file_offset
    }

    /// Returns the current cursor position relative to the window start.
    #[must_use]
    pub fn position(&self) -> u64 {
        self.cursor
    }

    /// Opens the underlying file and reads the entire window contents.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] on file open or read failure.
    pub fn read_all(&self) -> io::Result<Vec<u8>> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.file_offset))?;
        let mut buf = vec![0u8; self.len as usize];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Reads bytes from the current cursor position into `buf`.
    ///
    /// Opens a fresh file handle on each call; prefer [`FileWindow::read_all`]
    /// for bulk reads.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] on file open or read failure.
    pub fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let remaining = self.len.saturating_sub(self.cursor);
        if remaining == 0 {
            return Ok(0);
        }
        let to_read = (buf.len() as u64).min(remaining) as usize;
        let abs_offset = self.file_offset + self.cursor;
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(abs_offset))?;
        let n = file.read(&mut buf[..to_read])?;
        self.cursor += n as u64;
        Ok(n)
    }

    /// Seeks the window cursor.
    ///
    /// # Errors
    ///
    /// Returns an error if the resulting position would be negative.
    pub fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let window_len = self.len as i64;
        let new_cursor: i64 = match pos {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::End(delta) => window_len + delta,
            SeekFrom::Current(delta) => self.cursor as i64 + delta,
        };
        if new_cursor < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "FileWindow seek before beginning",
            ));
        }
        self.cursor = new_cursor as u64;
        Ok(self.cursor)
    }
}

// ─── MP4 box probe helper ─────────────────────────────────────────────────────

/// Probes a 4-byte big-endian MP4/MOV box at `offset` using the reader,
/// returning `(size, box_type_fourcc)`.
///
/// Handles the 64-bit extended size (`size == 1`) automatically.
///
/// # Errors
///
/// Returns an [`io::Error`] on read failure or truncated data.
pub fn probe_mp4_box(reader: &mut LargeFileReader, offset: u64) -> io::Result<(u64, [u8; 4])> {
    let size_u32 = reader.read_u32_be(offset)?;
    let box_type: [u8; 4] = reader.read_array_at(offset + 4)?;

    let actual_size: u64 = if size_u32 == 1 {
        reader.read_u64_be(offset + 8)?
    } else {
        size_u32 as u64
    };

    Ok((actual_size, box_type))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file_with_data(data: &[u8]) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!("oximedia_lfr_test_{id}.bin"));
        let mut f = File::create(&path).expect("create temp file");
        f.write_all(data).expect("write temp data");
        // Ensure the file is flushed and metadata is visible.
        f.flush().expect("flush");
        drop(f);
        path
    }

    #[test]
    fn test_open_and_file_len() {
        let data: Vec<u8> = (0u8..=255).collect();
        let path = temp_file_with_data(&data);
        let reader = LargeFileReader::open_default(&path).expect("open");
        assert_eq!(reader.file_len(), 256);
        assert!(!reader.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_at_basic() {
        let data: Vec<u8> = (0u8..=9).collect();
        let path = temp_file_with_data(&data);
        let mut reader = LargeFileReader::open_default(&path).expect("open");
        let mut buf = [0u8; 5];
        let n = reader.read_at(0, &mut buf).expect("read_at");
        assert_eq!(n, 5);
        assert_eq!(&buf, &[0, 1, 2, 3, 4]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_at_offset() {
        let data: Vec<u8> = (0u8..16).collect();
        let path = temp_file_with_data(&data);
        let mut reader = LargeFileReader::open_default(&path).expect("open");
        let mut buf = [0u8; 4];
        reader.read_at(8, &mut buf).expect("read_at offset");
        assert_eq!(&buf, &[8, 9, 10, 11]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_exact_at() {
        let data: Vec<u8> = (0u8..32).collect();
        let path = temp_file_with_data(&data);
        let mut reader = LargeFileReader::open_default(&path).expect("open");
        let mut buf = [0u8; 4];
        reader.read_exact_at(4, &mut buf).expect("read_exact_at");
        assert_eq!(&buf, &[4, 5, 6, 7]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_array_at() {
        let data = vec![0xDE_u8, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0];
        let path = temp_file_with_data(&data);
        let mut reader = LargeFileReader::open_default(&path).expect("open");
        let arr: [u8; 4] = reader.read_array_at(0).expect("read array");
        assert_eq!(arr, [0xDE, 0xAD, 0xBE, 0xEF]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_u32_be() {
        let val: u32 = 0x0102_0304;
        let data = val.to_be_bytes().to_vec();
        let path = temp_file_with_data(&data);
        let mut reader = LargeFileReader::open_default(&path).expect("open");
        let result = reader.read_u32_be(0).expect("read_u32_be");
        assert_eq!(result, val);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_stats_tracking() {
        let data = vec![0u8; 64];
        let path = temp_file_with_data(&data);
        let mut reader = LargeFileReader::open_default(&path).expect("open");
        let mut buf = [0u8; 16];
        reader.read_at(0, &mut buf).expect("read 1");
        reader.read_at(16, &mut buf).expect("read 2");
        let stats = reader.stats();
        assert_eq!(stats.read_calls, 2);
        assert_eq!(stats.bytes_read, 32);
        assert_eq!(stats.peak_read_bytes, 16);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_window_basic() {
        let data: Vec<u8> = (0u8..32).collect();
        let path = temp_file_with_data(&data);
        let reader = LargeFileReader::open_default(&path).expect("open");
        let win = reader.window(8, 8).expect("window");
        assert_eq!(win.len(), 8);
        assert_eq!(win.file_offset(), 8);
        let contents = win.read_all().expect("read_all");
        assert_eq!(contents, vec![8, 9, 10, 11, 12, 13, 14, 15]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_window_out_of_bounds_errors() {
        let data = vec![0u8; 16];
        let path = temp_file_with_data(&data);
        let reader = LargeFileReader::open_default(&path).expect("open");
        assert!(reader.window(8, 16).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_probe_mp4_box_basic() {
        // ftyp box: size=12, type="ftyp"
        let mut data = vec![0u8; 32];
        data[0] = 0;
        data[1] = 0;
        data[2] = 0;
        data[3] = 12; // size = 12
        data[4] = b'f';
        data[5] = b't';
        data[6] = b'y';
        data[7] = b'p';
        let path = temp_file_with_data(&data);
        let mut reader = LargeFileReader::open_default(&path).expect("open");
        let (size, box_type) = probe_mp4_box(&mut reader, 0).expect("probe");
        assert_eq!(size, 12);
        assert_eq!(&box_type, b"ftyp");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_config_sequential() {
        let cfg = LargeFileConfig::sequential();
        assert_eq!(cfg.access_pattern, AccessPattern::Sequential);
        assert!(cfg.buffer_size >= 1024 * 1024);
    }

    #[test]
    fn test_config_random_access() {
        let cfg = LargeFileConfig::random_access();
        assert_eq!(cfg.access_pattern, AccessPattern::Random);
        assert_eq!(cfg.prefetch_bytes, 0);
    }

    #[test]
    fn test_config_with_buffer_size() {
        let cfg = LargeFileConfig::default().with_buffer_size(1024);
        assert_eq!(cfg.buffer_size, 1024);
    }

    #[test]
    fn test_read_stats_mean_read_bytes() {
        let stats = ReadStats {
            bytes_read: 200,
            read_calls: 4,
            seek_count: 2,
            peak_read_bytes: 80,
        };
        assert!((stats.mean_read_bytes() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_contains_offset() {
        let data = vec![0u8; 100];
        let path = temp_file_with_data(&data);
        let reader = LargeFileReader::open_default(&path).expect("open");
        assert!(reader.contains_offset(0));
        assert!(reader.contains_offset(99));
        assert!(!reader.contains_offset(100));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_window_seek() {
        let data: Vec<u8> = (0u8..32).collect();
        let path = temp_file_with_data(&data);
        let reader = LargeFileReader::open_default(&path).expect("open");
        let mut win = reader.window(0, 16).expect("window");
        win.seek(SeekFrom::Start(4)).expect("seek");
        assert_eq!(win.position(), 4);
        let _ = std::fs::remove_file(&path);
    }
}
